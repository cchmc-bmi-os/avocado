require.def('design/form', [], {
    
    Form : function(view, concept_pk){
          var $form = $('<span class="form_container"></span>');
          var decOperatorsTmpl =  ['<option selected id="<%=this.field_id%>" value="range">is between</option>',
                                   '<option id="<%=this.field_id%>" value="-range">is not between</option>',
                                   '<option id="<%=this.field_id%>" value="lt">is less than</option>',
                                   '<option id="<%=this.field_id%>" value="gt">is greater than</option>',
                                   '<option id="<%=this.field_id%>" value="lte">is less than or equal to</option>',
                                   '<option id="<%=this.field_id%>" value="gte">is greater than or equal to</option>',
                                   '<option id="<%=this.field_id%>" value="exact">is equal to</option>',
                                   '<option id="<%=this.field_id%>" value="-exact">is not equal to</option>',
                                   '<option id="<%=this.field_id%>" value="isnull">is null</option>',
                                   '<option id="<%=this.field_id%>" value="-isnull">is not null</option>'].join('');
                                   
          var choiceOperatorsTmpl = ['<option selected value="in">is equal to</option>',
                                     '<option value="-in">is not equal to</option>'].join('');
                                     
          // For most cases we use the name attribute to constuct a unique id for all inputs (see field_id in the template context 
          // object below). The format for it is <concept primary key>_<field primary key> with optional "_input[01]" to support datatypes that
          // require ranges, and "_operator" to indicate the field represents an operator that can be changed by the users. With nothing appended to the 
          // end of the name, this indicates a "choice" or "assertion" (which maps to a check box) datatype. There is one exception that complicates things:
          // Some concepts include a field that is itself variable. That is to say that the user will supply some sort of value, for example, a decimal range,
          // and then they will need to tell us which field in the database this range should be applied to. An example should clear this up. For Pure
          // Tone Average (PTA), the user will select a range in the graph, and then below there will be a drop box that says "in the" and then a single choice
          // dropdown. The choices in the dropdown ("both","better ear", "worse ear") will actually represent the fields that the PTA ranges are to be applied to.
          // To indicate this type of a field, the concept description from the server will not contain a "pk" field for these types of fields. This will indicate 
          // that the pk is determined by the user. Fields that whose database field id are "variable" will have their name attribute set like this 
          // <concept id>_tbd
          
          $.each(view.fields, function(index,element){
              var input = []; // avoid odd exception if the server sends nothing
              
              switch (element.datatype) {
                  case 'boolean':  input = ['<label for="<%=this.field_id%>"><%=this.label%></label>',
                                            '<select id ="<%=this.field_id%>" name="<%=this.field_id%>">',
                                                '<option value="Yes">Yes</option>',
                                                '<option value="No">No</option>',
                                            '</select>'];
                                   break;
                  case 'nullboolean': input = ['<label for="<%=this.field_id%>"><%=this.label%></label>',
                                               '<select id ="<%=this.field_id%>" name="<%=this.field_id%>">',
                                                  '<option value="Yes">Yes</option>',
                                                  '<option value="No">No</option>',
                                               '</select>'];
                                      break;
                  case 'assertion':input = ['<input type="checkbox" name="<%=this.field_id%>" value="<%=this.field_id%>" <%= this["default"] ? "checked":""%>/>',
                                            '<label for="<%=this.field_id%>"><%=this.label%></label>'];
                                   break;
                  case 'number'  :input = ['<label for="<%=this.field_id%>"><%=this.label%></label>',
                                            '<select id="<%=this.field_id%>_operator" name="<%=this.field_id%>_operator">',
                                               decOperatorsTmpl,
                                            '</select>',
                                            '<span class="input_association" name="<%=this.field_id%>_input_assoc">',
                                            '<input data-validate="decimal" id="<%=this.field_id%>_input0" type="text" name="<%=this.field_id%>_input0" size="5">',
                                            '<label for="<%=this.field_id%>_input1">and</label>',
                                            '<input data-validate="decimal" id="<%=this.field_id%>_input1" type="text" name="<%=this.field_id%>_input1" size="5">',
                                            '</span>'];
                                   break;
                 case 'string'    : input = [ '<label for="<%=this.field_id%>"><%=this.label%></label>', // No defaults for this type, doesn't make sense
                                              '<select id="<%=this.field_id%>-operator" name="<%=this.field_id%>_operator">',
                                                 choiceOperatorsTmpl,
                                              '</select>',
                                              '<select multiple="multiple" id="<%=this.field_id%>-value" name="<%=this.field_id%>" size="3" data-optional="<%=this.optional%>" >',
                                              '<% for (index in this.choices) { %>',
                                                    '<option value="<%=this.choices[index][0]%>"><%=this.choices[index][1]%></option>',
                                               '<%}%>',
                                              '</select>'
                                            ];
                                    break;
               }
               
               
               // Wrap each discrete element in <p>
               input.unshift("<p>");
               
               input.push("</p>");
               // Does this element contain a "pk" attribute? See large comment above for reason
               if (!element.hasOwnProperty("pk")){
                   // Append additional dropdown for user to choose which field this applies to
                    input = input.concat(['<p><label for="<%=this.pkchoice_id%>"><%=this.pkchoice_label%></label>',
                                  '<select id="<%=this.pkchoice_id%>" name="<%=this.pkchoice_id%>">',
                                  '<% for (index in this.pkchoices) { %>',     
                                          '<option value="<%=this.pkchoices[index][0]%>" <%=this.pkchoices[index][0]==this.pkchoice_default ? "selected":""%>><%=this.pkchoices[index][1]%></option>',
                                  '<%}%>',
                                  '</select></p>']);
               }
               
               // This should come out, the server should send us No Data instead of null
               $.each(['choices', 'pkchoices'], function(index, attr){
                    element[attr] && $.each(element[attr], function(index, choice){
                          // null needs to be "No Data"
                          if ( element[attr][index][0] === null){
                               element[attr][index][0] = "No Data";
                          }          
                          if ( element[attr][index][1] === "None"){
                               element[attr][index][1] = "No Data";
                          }           
                    });
                });
                
                // The following scheme for generating name attributes will be used:
                // Elements that represent non-variable fields on the concept will have their name attribute constructed as follows:
                // <concept_id>_<field_id>
                // Elements that can be applied to a variable primary key will have their name attribute constructed as follows:
                // <concept_id>_<sorted list of possible field_ids separated by 'OR'>
                // Elements that represent the dropdown operator which will be used to determine the variable field ID will have their
                // name attribute constructed as follows
                // <sorted list of possible field_id choices contained within separated by 'OR'> 
                // This should be the same string that comes after the <concept_id> in the element that is dependent on this one
                
                var name_attribute = null;
                var pkchoice_name_attribute = null;
                if (element.hasOwnProperty("pk")){
                    name_attribute = concept_pk+"_"+element.pk;
                }else{
                    var int_ids = $.map(element.pkchoices, function(element, index){
                        return parseInt(element[0]);
                    });
                    int_ids.sort();
                    pkchoice_name_attribute = int_ids.join("OR");
                    name_attribute = concept_pk + "_" + pkchoice_name_attribute;
                }
                
                var $row = $($.jqote(input.join(""), {"choices":element.choices,
                                                      "field_id":name_attribute,
                                                      "label":element.name,
                                                      "pkchoices":element.pkchoices,
                                                      "pkchoice_label":element.pkchoice_label,
                                                      "pkchoice_id":pkchoice_name_attribute,
                                                      "optional": element.hasOwnProperty('optional') ? element.optional : false,
                                                      "default": element.hasOwnProperty('default') ? element["default"]: 0,
                                                      "pkchoice_default": element.hasOwnProperty('pkchoice_default') ? element["pkchoice_default"]: 0}));
                $row.children().not("span").wrap("<span/>");
                $form.append($row);
         });
         
         // Make form into a table
         //$("label,select,input,span", $form).not("span *").wrap("<td/>");
         //$("p", $form).wrap("<tr/>");
         //$("p",$form).children().unwrap();
         
         // Trigger an event when anything changes
         $("input,select",$form).bind('change keyup', function(evt){
            var $target = $(evt.target); 
            switch (evt.target.type){
                    case "checkbox": $form.trigger("ElementChangedEvent", [{name:evt.target.name,value:evt.target.checked}]);
                                     break;
                    case "select-one"      :
                    case "select-multiple" : 
                    case "select"          : var selected = []; 
                                              $("option", $(evt.target)).each(function(index,opt){
                                                 if  (opt.selected){
                                                     selected.push(opt.value);
                                                     // Do we need to show 1, 2, or no inputs?
                                                     
                                                    if (opt.value.search(/range/) >= 0){
                                                         $("input[name="+opt.id+"_input1]",$form).show().change();
                                                         $("label[for="+opt.id+"_input1]",$form).show();
                                                         // Trigger change on associated inputs because they need to work with a range operator now
                                                         $("input[name="+opt.id+"_input0]",$form).show().change();
                                                     } else if (opt.value.search(/null/) >= 0){
                                                         $("input[name="+opt.id+"_input1],",$form).hide().change();
                                                         $("label[for="+opt.id+"_input1]",$form).hide();
                                                         // Trigger change on associated inputs because they need to work with a range operator now
                                                         $("input[name="+opt.id+"_input0]",$form).hide().change();
                                                             
                                                     } else {
                                                         $("input[name="+opt.id+"_input1],",$form).hide().change();
                                                         $("label[for="+opt.id+"_input1]",$form).hide();
                                                         // Trigger change on associated inputs because they need to work with a range operator now
                                                         $("input[name="+opt.id+"_input0]",$form).show().change();
                                                     }
                                                 }
                                             });
                                             // Since this code executes for select choices boxes as well as operators (which should
                                             // never be plural), we make sure to send the correct type array, or single item
                                             var sendValue;
                                             if (evt.target.type === "select-multiple"){
                                                 // If a select-multiple box is optional, and nothing is selected, send null so that it doesn't appear as empty in 
                                                 // the datasource, eitherwise, we will throw an error if nothing is supplied;
                                                 if ($target.is('[data-optional=true]')) {
                                                    sendValue = selected.length ? selected : null;
                                                 } else {
                                                    sendValue = selected; 
                                                 }
                                             } else { 
                                                 sendValue = selected[0]; // JMM I am concerned selected[0] could sometimes be undefined?
                                             }
                                            
                                             
                                             $form.trigger("ElementChangedEvent", [{name:evt.target.name, value:sendValue}]);
                                             break;
                    default   : // This catches input boxes, if input boxes are not currently visible, send null for them
                                // Input boxes require an extra validation step because of the free form input
                                
                                var associated_operator = $(evt.target).closest("p").find("select").val();
                                var name_prefix = evt.target.name.substr(0,evt.target.name.length-1);
                                var $input1 = $("input[name="+name_prefix+"0]",$form);
                                var $input2 = $("input[name="+name_prefix+"1]",$form);
                                var value1 = parseFloat($input1.val());
                                var value2 = parseFloat($input2.val());
                                
                                // This one is a little tricky because it matters not just that the fields map to valid numbers
                                // but that in the case of a range style operator, the two numbers are sequential, and finally
                                // if fields have become hidden due to a change in operator, we no longer want to list that something
                                // is wrong with the field even if there is (because it doesn't matter)
                                switch ($target.attr('data-validate')){
                                    case "decimal": if ($target.is(":visible") && isNaN(Number($target.val()))) {
                                                        // Field contains a non-number and is visible
                                                        var input_evt = $.Event("InvalidInputEvent");
                                                        $target.trigger(input_evt);
                                                    }else if ($(evt.target).hasClass('invalid')){ //TODO Don't rely on this
                                                        // Field either contains a number or is not visible
                                                        // Either way it was previously invalid
                                                        input_evt = $.Event("InputCorrectedEvent");
                                                        $target.trigger(input_evt);
                                                    } else if ((associated_operator.search(/range/) >= 0) && (value1 > value2)) { // && ($input2.is(":visible"))) {
                                                        // A range operator is in use, both fields are visible, and their values are not sequential
                                                        input_evt = $.Event("InvalidInputEvent");
                                                        input_evt.reason = "badrange";
                                                        input_evt.message = "First input must be less than second input.";
                                                        $target.parent().trigger(input_evt);
                                                    } else if ($(evt.target).parent().hasClass('invalid_badrange') && (($input2.css("display") === "none")||(value1 < value2))){ //TODO Don't rely on this
                                                        // A range operator is or was in use, and either a range operator is no longer in use, so we don't care, or 
                                                        // its in use but the values are now sequential.
                                                        input_evt = $.Event("InputCorrectedEvent");
                                                        input_evt.reason = "badrange";
                                                        $target.parent().trigger(input_evt);
                                                    }   
                                                    break;
                                    default: break;
                                }
                                var value = $target.is(":visible") ? $target.val() : null;
                                $form.trigger("ElementChangedEvent", [{name:evt.target.name,value:value}]);
                                break;
             }
             evt.stopPropagation();
         });
         
         var updateElement = function(evt, element) {
                 var $element = $("[name="+element.name+"]", $form);
                 // Note: Just because we are here doesn't mean we contain the element
                 // to be updated, it may reside on another view within this concept
                 if ($element.length === 0) return;
                 var type = $element.attr("type");
                 switch (type){
                     case "checkbox": $element.attr("checked",element.value);
                                      break;
                     case "select-multiple" : $("option", $element).each(function(index,opt){
                                                  if ($.inArray(opt.value,element.value)!=-1){
                                                      opt.selected = true;
                                                  }else{
                                                      opt.selected = false;
                                                  }
                                              });
                                              break; 
                     default:   // inputs and singular selects 
                                $element.attr("value",element.value); 
                                break;
                 }
                 
         };
         
         $form.bind("UpdateElementEvent", updateElement);
         
         $form.bind("UpdateDSEvent", function(evt, ds){
             for (var key in ds){
                 updateElement(null, {name:key, value:ds[key]});
             }
         });
         
         $form.bind("GainedFocusEvent", function(evt) {
             $("input,select", $form).change(); 
         });
         return $form;
    }
});